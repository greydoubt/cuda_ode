// math_function.cuh
#ifndef MATHFUNCTION_H_
#define MATHFUNCTION_H_
template <class Res>
class MathFunction {
public:
        MathFunction();
        virtual ~MathFunction();
        virtual Res operator()(Res value) = 0;
};
#endif 
